/*
 * parsim-omp.c - Implementação OpenMP do simulador de partículas
 * Versão corrigida para resultados consistentes com a serial
 * 
 * Compilar com: gcc -fopenmp -O0 -o parsim-omp parsim-omp.c -lm
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <math.h>
 #include <omp.h>
 
 #define G 6.67408e-11
 #define EPSILON 0.005
 #define EPSILON2 (EPSILON*EPSILON)
 #define DELTAT 0.1
 
 typedef struct {
     double x, y;
     double vx, vy;
     double m;
     int active;
 } particle_t;
 
 typedef struct {
     double mass;
     double com_x;
     double com_y;
 } cell_t;
 
 unsigned int seed_global;
 
 void init_r4uni(int input_seed) {
     seed_global = input_seed + 987654321;
 }
 
 double rnd_uniform01() {
     int seed_in = seed_global;
     seed_global ^= (seed_global << 13);
     seed_global ^= (seed_global >> 17);
     seed_global ^= (seed_global << 5);
     return 0.5 + 0.2328306e-09 * (seed_in + (int)seed_global);
 }
 
 double rnd_normal01() {
     double u1, u2, z, result;
     do {
         u1 = rnd_uniform01();
         u2 = rnd_uniform01();
         z = sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
         result = 0.5 + 0.15 * z;
     } while (result < 0 || result >= 1);
     return result;
 }
 
 void init_particles(long seed, double side, long ncside, long long n_part, particle_t *par) {
     double (*rnd01)() = rnd_uniform01;
     if (seed < 0) {
         rnd01 = rnd_normal01;
         seed = -seed;
     }
     init_r4uni((int)seed);
     
     for (long long i = 0; i < n_part; i++) {
         par[i].x = rnd01() * side;
         par[i].y = rnd01() * side;
         par[i].vx = (rnd01() - 0.5) * side / ncside / 5.0;
         par[i].vy = (rnd01() - 0.5) * side / ncside / 5.0;
         par[i].m = rnd01() * 0.01 * ((double)(ncside * ncside)) / n_part / G * EPSILON2;
         par[i].active = 1;
     }
 }
 
 int mod(int a, int b) {
     int r = a % b;
     return r < 0 ? r + b : r;
 }
 
 int get_cell_index(double x, double y, double side, int ncside) {
     double cell_size = side / ncside;
     int cx = mod((int)(x / cell_size), ncside);
     int cy = mod((int)(y / cell_size), ncside);
     return cy * ncside + cx;
 }
 
 double periodic_diff(double d, double side) {
     if (d > side/2) return d - side;
     if (d < -side/2) return d + side;
     return d;
 }
 
 void simulation(particle_t *par, long long n_part, double side, int ncside, int n_steps, int *collision_count) {
     int step, i, j, ix, iy;
     double cell_size = side / ncside;
     int n_cells = ncside * ncside;
     cell_t *cells = (cell_t*)malloc(n_cells * sizeof(cell_t));
     *collision_count = 0;
 
     // Variáveis para OpenMP
     double fx, fy;
     int cell_idx, cx, cy, ncx, ncy, ncell;
     double dx, dy, dist2, force, dist;
 
     for (step = 0; step < n_steps; step++) {
         // 1. Zerar células
         #pragma omp parallel for
         for (i = 0; i < n_cells; i++) {
             cells[i].mass = 0.0;
             cells[i].com_x = 0.0;
             cells[i].com_y = 0.0;
         }
 
         // 2. Atribuir partículas às células
         #pragma omp parallel for
         for (i = 0; i < n_part; i++) {
             if (!par[i].active) continue;
             int idx = get_cell_index(par[i].x, par[i].y, side, ncside);
             #pragma omp atomic
             cells[idx].mass += par[i].m;
             #pragma omp atomic
             cells[idx].com_x += par[i].m * par[i].x;
             #pragma omp atomic
             cells[idx].com_y += par[i].m * par[i].y;
         }
 
         // 3. Calcular centros de massa
         #pragma omp parallel for
         for (i = 0; i < n_cells; i++) {
             if (cells[i].mass > 0) {
                 cells[i].com_x /= cells[i].mass;
                 cells[i].com_y /= cells[i].mass;
             }
         }
 
         // 4. Calcular forças
         #pragma omp parallel for private(fx, fy, cell_idx, cx, cy, ncx, ncy, ncell, dx, dy, dist2, force, dist)
         for (i = 0; i < n_part; i++) {
             if (!par[i].active) continue;
             fx = 0.0;
             fy = 0.0;
             cell_idx = get_cell_index(par[i].x, par[i].y, side, ncside);
             cx = cell_idx % ncside;
             cy = cell_idx / ncside;
 
             for (iy = -1; iy <= 1; iy++) {
                 for (ix = -1; ix <= 1; ix++) {
                     ncx = mod(cx + ix, ncside);
                     ncy = mod(cy + iy, ncside);
                     ncell = ncy * ncside + ncx;
 
                     if (ncx == cx && ncy == cy) {
                         for (j = 0; j < n_part; j++) {
                             if (!par[j].active || j == i) continue;
                             if (get_cell_index(par[j].x, par[j].y, side, ncside) == cell_idx) {
                                 dx = periodic_diff(par[j].x - par[i].x, side);
                                 dy = periodic_diff(par[j].y - par[i].y, side);
                                 dist2 = dx*dx + dy*dy;
                                 if (dist2 > 1e-12) {
                                     force = G * par[i].m * par[j].m / dist2;
                                     dist = sqrt(dist2);
                                     fx += force * dx / dist;
                                     fy += force * dy / dist;
                                 }
                             }
                         }
                     } else if (cells[ncell].mass > 0) {
                         dx = periodic_diff(cells[ncell].com_x - par[i].x, side);
                         dy = periodic_diff(cells[ncell].com_y - par[i].y, side);
                         dist2 = dx*dx + dy*dy;
                         if (dist2 > 1e-12) {
                             force = G * par[i].m * cells[ncell].mass / dist2;
                             dist = sqrt(dist2);
                             fx += force * dx / dist;
                             fy += force * dy / dist;
                         }
                     }
                 }
             }
             par[i].vx += (fx / par[i].m) * DELTAT;
             par[i].vy += (fy / par[i].m) * DELTAT;
         }
 
         // 5. Atualizar posições
         #pragma omp parallel for
         for (i = 0; i < n_part; i++) {
             if (!par[i].active) continue;
             par[i].x += par[i].vx * DELTAT;
             par[i].y += par[i].vy * DELTAT;
             
             if (par[i].x < 0) par[i].x += side;
             if (par[i].x >= side) par[i].x -= side;
             if (par[i].y < 0) par[i].y += side;
             if (par[i].y >= side) par[i].y -= side;
         }
 
         // 6. Verificação de colisões - Versão consistente com a serial
         int local_collisions = 0;
         #pragma omp parallel for reduction(+:local_collisions)
         for (i = 0; i < n_part; i++) {
             if (!par[i].active) continue;
             int cell_i = get_cell_index(par[i].x, par[i].y, side, ncside);
             
             for (j = i+1; j < n_part; j++) {
                 if (!par[j].active) continue;
                 if (get_cell_index(par[j].x, par[j].y, side, ncside) != cell_i) continue;
                 
                 double dx = periodic_diff(par[i].x - par[j].x, side);
                 double dy = periodic_diff(par[i].y - par[j].y, side);
                 double dist2 = dx*dx + dy*dy;
                 
                 if (dist2 < EPSILON2) {
                     #pragma omp critical
                     {
                         // Verificação redundante para garantir consistência
                         if (par[i].active && par[j].active) {
                             par[i].active = 0;
                             par[j].active = 0;
                             local_collisions += 2;
                         }
                     }
                     break; // Importante para evitar múltiplas colisões
                 }
             }
         }
         *collision_count += local_collisions;
     }
     free(cells);
 }
 
 int main(int argc, char *argv[]) {
     if (argc != 6) {
         fprintf(stderr, "Uso: %s <semente> <lado> <ncside> <n_part> <n_steps>\n", argv[0]);
         return EXIT_FAILURE;
     }
 
     long seed = atol(argv[1]);
     double side = atof(argv[2]);
     int ncside = atoi(argv[3]);
     long long n_part = atoll(argv[4]);
     int n_steps = atoi(argv[5]);
 
     particle_t *particles = (particle_t*)malloc(n_part * sizeof(particle_t));
     if (!particles) {
         fprintf(stderr, "Erro ao alocar partículas\n");
         return EXIT_FAILURE;
     }
 
     init_particles(seed, side, ncside, n_part, particles);
 
     double exec_time = -omp_get_wtime();
     int collision_count = 0;
     simulation(particles, n_part, side, ncside, n_steps, &collision_count);
     exec_time += omp_get_wtime();
 
     fprintf(stderr, "%.1fs\n", exec_time);
     printf("%.3f %.3f\n", particles[0].x, particles[0].y);
     printf("%d\n", collision_count);
 
     free(particles);
     return EXIT_SUCCESS;
 }