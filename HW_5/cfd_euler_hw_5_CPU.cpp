#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <sstream>
#include <chrono>
#include <omp.h>

using namespace std;

// ------------------------------------------------------------
// Global parameters
// ------------------------------------------------------------
const double gamma_val = 1.4;   // Ratio of specific heats
const double CFL = 0.5;         // CFL number

// ------------------------------------------------------------
// Compute pressure from the conservative variables
// ------------------------------------------------------------
double pressure(double rho, double rhou, double rhov, double E) {
    double u = rhou / rho;
    double v = rhov / rho;
    double kinetic = 0.5 * rho * (u * u + v * v);
    return (gamma_val - 1.0) * (E - kinetic);
}

// ------------------------------------------------------------
// Compute flux in the x-direction
// ------------------------------------------------------------
void fluxX(double rho, double rhou, double rhov, double E, 
           double& frho, double& frhou, double& frhov, double& fE) {
    double u = rhou / rho;
    double p = pressure(rho, rhou, rhov, E);
    frho = rhou;
    frhou = rhou * u + p;
    frhov = rhov * u;
    fE = (E + p) * u;
}

// ------------------------------------------------------------
// Compute flux in the y-direction
// ------------------------------------------------------------
void fluxY(double rho, double rhou, double rhov, double E,
           double& frho, double& frhou, double& frhov, double& fE) {
    double v = rhov / rho;
    double p = pressure(rho, rhou, rhov, E);
    frho = rhov;
    frhou = rhou * v;
    frhov = rhov * v + p;
    fE = (E + p) * v;
}

// ------------------------------------------------------------
// Main simulation routine
// ------------------------------------------------------------
int main(){
    // ----- Grid and domain parameters -----
    const int Nx = 800;         // Number of cells in x (excluding ghost cells)
    const int Ny = 400;         // Number of cells in y
    const double Lx = 2.0;      // Domain length in x
    const double Ly = 1.0;      // Domain length in y
    const double dx = Lx / Nx;
    const double dy = Ly / Ny;

    cout << "Initializing CPU simulation with Grid Size: " << Nx << " x " << Ny << endl;
    cout << "Max OpenMP threads available: " << omp_get_max_threads() << endl;

    // Create flat arrays (with ghost cells)
    const int total_size = (Nx + 2) * (Ny + 2);
    
    double* rho = (double*)malloc(total_size * sizeof(double));
    double* rhou = (double*)malloc(total_size * sizeof(double));
    double* rhov = (double*)malloc(total_size * sizeof(double));
    double* E = (double*)malloc(total_size * sizeof(double));
    double* rho_new = (double*)malloc(total_size * sizeof(double));
    double* rhou_new = (double*)malloc(total_size * sizeof(double));
    double* rhov_new = (double*)malloc(total_size * sizeof(double));
    double* E_new = (double*)malloc(total_size * sizeof(double));
    bool* solid = (bool*)malloc(total_size * sizeof(bool));

    #pragma omp parallel for
    for (int i = 0; i < total_size; i++) {
      rho[i] = 0.0; rhou[i] = 0.0; rhov[i] = 0.0; E[i] = 0.0;
      rho_new[i] = 0.0; rhou_new[i] = 0.0; rhov_new[i] = 0.0; E_new[i] = 0.0;
      solid[i] = false;
    }

    // ----- Obstacle (cylinder) parameters -----
    const double cx = 0.5;      
    const double cy = 0.5;      
    const double radius = 0.1;  

    // ----- Free-stream initial conditions (inflow) -----
    const double rho0 = 1.0;
    const double u0 = 1.0;
    const double v0 = 0.0;
    const double p0 = 1.0;
    const double E0 = p0/(gamma_val - 1.0) + 0.5*rho0*(u0*u0 + v0*v0);

    long long n_solid = 0;
    long long n_interior_update = 0;

    // ----- Initialize grid and obstacle mask -----
    for (int i = 0; i < Nx+2; i++){
        for (int j = 0; j < Ny+2; j++){
            double x = (i - 0.5) * dx;
            double y = (j - 0.5) * dy;
            if ((x - cx)*(x - cx) + (y - cy)*(y - cy) <= radius * radius) {
                solid[i*(Ny+2)+j] = true;
                rho[i*(Ny+2)+j] = rho0;
                rhou[i*(Ny+2)+j] = 0.0;
                rhov[i*(Ny+2)+j] = 0.0;
                E[i*(Ny+2)+j] = p0/(gamma_val - 1.0);
            } else {
                solid[i*(Ny+2)+j] = false;
                rho[i*(Ny+2)+j] = rho0;
                rhou[i*(Ny+2)+j] = rho0 * u0;
                rhov[i*(Ny+2)+j] = rho0 * v0;
                E[i*(Ny+2)+j] = E0;
            }
        }
    }

    for (int i = 1; i <= Nx; i++){
        for (int j = 1; j <= Ny; j++){
            if (solid[i*(Ny+2)+j]) n_solid++;
            else n_interior_update++;
        }
    }

    // ----- Determine time step from CFL condition -----
    double c0 = sqrt(gamma_val * p0 / rho0);
    double dt = CFL * min(dx, dy) / (fabs(u0) + c0)/2.0;

    const int nSteps = 2000;
    
    // Performance tracking variables
    double boundary_time = 0.0, interior_time = 0.0, copy_time = 0.0, energy_time = 0.0;
    int boundary_loops = 0, interior_loops = 0, copy_loops = 0, energy_loops = 0;

    auto main_start = std::chrono::high_resolution_clock::now();

    // ----- Main time-stepping loop -----
    for (int n = 0; n < nSteps; n++){
        
        // --- Apply boundary conditions on ghost cells ---
        auto boundary_start = std::chrono::high_resolution_clock::now();
        
        #pragma omp parallel for
        for (int j = 0; j < Ny+2; j++){
            rho[0*(Ny+2)+j] = rho0;
            rhou[0*(Ny+2)+j] = rho0*u0;
            rhov[0*(Ny+2)+j] = rho0*v0;
            E[0*(Ny+2)+j] = E0;
        }
        #pragma omp parallel for
        for (int j = 0; j < Ny+2; j++){
            rho[(Nx+1)*(Ny+2)+j] = rho[Nx*(Ny+2)+j];
            rhou[(Nx+1)*(Ny+2)+j] = rhou[Nx*(Ny+2)+j];
            rhov[(Nx+1)*(Ny+2)+j] = rhov[Nx*(Ny+2)+j];
            E[(Nx+1)*(Ny+2)+j] = E[Nx*(Ny+2)+j];
        }
        #pragma omp parallel for
        for (int i = 0; i < Nx+2; i++){
            rho[i*(Ny+2)+0] = rho[i*(Ny+2)+1];
            rhou[i*(Ny+2)+0] = rhou[i*(Ny+2)+1];
            rhov[i*(Ny+2)+0] = -rhov[i*(Ny+2)+1];
            E[i*(Ny+2)+0] = E[i*(Ny+2)+1];
        }
        #pragma omp parallel for
        for (int i = 0; i < Nx+2; i++){
            rho[i*(Ny+2)+(Ny+1)] = rho[i*(Ny+2)+Ny];
            rhou[i*(Ny+2)+(Ny+1)] = rhou[i*(Ny+2)+Ny];
            rhov[i*(Ny+2)+(Ny+1)] = -rhov[i*(Ny+2)+Ny];
            E[i*(Ny+2)+(Ny+1)] = E[i*(Ny+2)+Ny];
        }
        auto boundary_end = std::chrono::high_resolution_clock::now();
        boundary_time += std::chrono::duration<double>(boundary_end - boundary_start).count();
        boundary_loops++;

        // --- Update interior cells using a Lax-Friedrichs scheme ---
        auto interior_start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 1; i <= Nx; i++){
            for (int j = 1; j <= Ny; j++){
                if (solid[i*(Ny+2)+j]) {
                    rho_new[i*(Ny+2)+j] = rho[i*(Ny+2)+j];
                    rhou_new[i*(Ny+2)+j] = rhou[i*(Ny+2)+j];
                    rhov_new[i*(Ny+2)+j] = rhov[i*(Ny+2)+j];
                    E_new[i*(Ny+2)+j] = E[i*(Ny+2)+j];
                    continue;
                }

                rho_new[i*(Ny+2)+j] = 0.25 * (rho[(i+1)*(Ny+2)+j] + rho[(i-1)*(Ny+2)+j] + 
                                             rho[i*(Ny+2)+(j+1)] + rho[i*(Ny+2)+(j-1)]);
                rhou_new[i*(Ny+2)+j] = 0.25 * (rhou[(i+1)*(Ny+2)+j] + rhou[(i-1)*(Ny+2)+j] + 
                                              rhou[i*(Ny+2)+(j+1)] + rhou[i*(Ny+2)+(j-1)]);
                rhov_new[i*(Ny+2)+j] = 0.25 * (rhov[(i+1)*(Ny+2)+j] + rhov[(i-1)*(Ny+2)+j] + 
                                              rhov[i*(Ny+2)+(j+1)] + rhov[i*(Ny+2)+(j-1)]);
                E_new[i*(Ny+2)+j] = 0.25 * (E[(i+1)*(Ny+2)+j] + E[(i-1)*(Ny+2)+j] + 
                                           E[i*(Ny+2)+(j+1)] + E[i*(Ny+2)+(j-1)]);

                double fx_rho1, fx_rhou1, fx_rhov1, fx_E1;
                double fx_rho2, fx_rhou2, fx_rhov2, fx_E2;
                double fy_rho1, fy_rhou1, fy_rhov1, fy_E1;
                double fy_rho2, fy_rhou2, fy_rhov2, fy_E2;

                fluxX(rho[(i+1)*(Ny+2)+j], rhou[(i+1)*(Ny+2)+j], rhov[(i+1)*(Ny+2)+j], E[(i+1)*(Ny+2)+j],
                      fx_rho1, fx_rhou1, fx_rhov1, fx_E1);
                fluxX(rho[(i-1)*(Ny+2)+j], rhou[(i-1)*(Ny+2)+j], rhov[(i-1)*(Ny+2)+j], E[(i-1)*(Ny+2)+j],
                      fx_rho2, fx_rhou2, fx_rhov2, fx_E2);
                fluxY(rho[i*(Ny+2)+(j+1)], rhou[i*(Ny+2)+(j+1)], rhov[i*(Ny+2)+(j+1)], E[i*(Ny+2)+(j+1)],
                      fy_rho1, fy_rhou1, fy_rhov1, fy_E1);
                fluxY(rho[i*(Ny+2)+(j-1)], rhou[i*(Ny+2)+(j-1)], rhov[i*(Ny+2)+(j-1)], E[i*(Ny+2)+(j-1)],
                      fy_rho2, fy_rhou2, fy_rhov2, fy_E2);

                double dtdx = dt / (2 * dx);
                double dtdy = dt / (2 * dy);
                
                rho_new[i*(Ny+2)+j] -= dtdx * (fx_rho1 - fx_rho2) + dtdy * (fy_rho1 - fy_rho2);
                rhou_new[i*(Ny+2)+j] -= dtdx * (fx_rhou1 - fx_rhou2) + dtdy * (fy_rhou1 - fy_rhou2);
                rhov_new[i*(Ny+2)+j] -= dtdx * (fx_rhov1 - fx_rhov2) + dtdy * (fy_rhov1 - fy_rhov2);
                E_new[i*(Ny+2)+j] -= dtdx * (fx_E1 - fx_E2) + dtdy * (fy_E1 - fy_E2);
            }
        }
        auto interior_end = std::chrono::high_resolution_clock::now();
        interior_time += std::chrono::duration<double>(interior_end - interior_start).count();
        interior_loops++;

        // Copy updated values back
        auto copy_start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 1; i <= Nx; i++){
            for (int j = 1; j <= Ny; j++){
                rho[i*(Ny+2)+j] = rho_new[i*(Ny+2)+j];
                rhou[i*(Ny+2)+j] = rhou_new[i*(Ny+2)+j];
                rhov[i*(Ny+2)+j] = rhov_new[i*(Ny+2)+j];
                E[i*(Ny+2)+j] = E_new[i*(Ny+2)+j];
            }
        }
        auto copy_end = std::chrono::high_resolution_clock::now();
        copy_time += std::chrono::duration<double>(copy_end - copy_start).count();
        copy_loops++;

        // Calculate total kinetic energy
        auto energy_start = std::chrono::high_resolution_clock::now();
        double total_kinetic = 0.0;
        #pragma omp parallel for collapse(2) reduction(+:total_kinetic) schedule(static)
        for (int i = 1; i <= Nx; i++) {
            for (int j = 1; j <= Ny; j++) {
                double u = rhou[i*(Ny+2)+j] / rho[i*(Ny+2)+j];
                double v = rhov[i*(Ny+2)+j] / rho[i*(Ny+2)+j];
                total_kinetic += 0.5 * rho[i*(Ny+2)+j] * (u * u + v * v);
            }
        }
        auto energy_end = std::chrono::high_resolution_clock::now();
        energy_time += std::chrono::duration<double>(energy_end - energy_start).count();
        energy_loops++;

        if (n % 50 == 0) {
            cout << "Step " << n << " completed, total kinetic energy: " << total_kinetic << endl;
        }
    }

    auto main_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(main_end - main_start).count();

    // --- Performance Table Output ---
    const long long total_BC_bytes = (Ny + 2) * 96LL + (Nx + 2) * 128LL;
    const long long total_copy_bytes = (long long)Nx * Ny * 64LL;
    const long long total_energy_bytes = (long long)Nx * Ny * 24LL;
    const long long total_interior_bytes_solid = n_solid * 64LL;
    const long long total_interior_bytes_non_solid = n_interior_update * 224LL;
    const long long total_interior_bytes = total_interior_bytes_solid + total_interior_bytes_non_solid;

    struct KernelData {
        string name;
        int count;
        double time;
        long long bytes_per_step;
    };
    
    vector<KernelData> kernels = {
        {"Boundary Conditions", boundary_loops, boundary_time, total_BC_bytes},
        {"Interior Update", interior_loops, interior_time, total_interior_bytes},
        {"Copy Values", copy_loops, copy_time, total_copy_bytes},
        {"Calculate Kinetic Energy", energy_loops, energy_time, total_energy_bytes}
    };

    cout << endl;
    cout << setw(30) << left << "Name" << setw(10) << right << "Count" 
         << setw(12) << "Time" << setw(10) << "GB/s" << endl;
    
    for (const auto& k : kernels) {
        double bandwidth = 0.0;
        if (k.time > 0.0) {
            bandwidth = (double)(k.bytes_per_step * (long long)k.count) / k.time / 1.0e9;
        }
        cout << setw(30) << left << k.name << setw(10) << right << k.count
             << setw(10) << fixed << setprecision(3) << k.time << "s"
             << setw(10) << setprecision(2) << bandwidth << endl;
    }
    cout << "\nTotal execution time: " << total_time << " seconds" << endl;

    // Optional File Output
    ofstream outFile("HW_5_results_CPU.txt", ios::app);
    outFile << "Grid Size: " << Nx << "x" << Ny << endl; 
    outFile << setw(30) << left << "Name" << setw(10) << right << "Count" 
            << setw(12) << "Time" << setw(10) << "GB/s" << endl;
    for (const auto& k : kernels) {
        double bandwidth = 0.0;
        if (k.time > 0.0) {
            bandwidth = (double)(k.bytes_per_step * (long long)k.count) / k.time / 1.0e9;
        }
        outFile << setw(30) << left << k.name << setw(10) << right << k.count
                << setw(10) << fixed << setprecision(3) << k.time << "s"
                << setw(10) << setprecision(2) << bandwidth << endl;
    }
    outFile << "Total execution time: " << total_time << " seconds\n" << endl;
    outFile.close();

    // Free memory
    free(rho); free(rhou); free(rhov); free(E);
    free(rho_new); free(rhou_new); free(rhov_new); free(E_new);
    free(solid);

    return 0;
}