#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define G 6.67408e-11
#define EPSILON2 (0.005 * 0.005)
#define DELTAT 0.1

typedef struct {
    double x, y;
    double vx, vy;
    double m;
    int active;
} Particle;

unsigned int seed;

void init_r4uni(int input_seed) {
    seed = input_seed + 987654321;
}

double rnd_uniform01() {
    int seed_in = seed;
    seed ^= (seed << 13);
    seed ^= (seed >> 17);
    seed ^= (seed << 5);
    return 0.5 + 0.2328306e-09 * (seed_in + (int) seed);
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

void init_particles(long seed, double side, long ncside, long long n_part, Particle *par) {
    double (*rnd01)() = rnd_uniform01;
    long long i;
    if (seed < 0) {
        rnd01 = rnd_normal01;
        seed = -seed;
    }
    init_r4uni(seed);
    for (i = 0; i < n_part; i++) {
        par[i].x = rnd01() * side;
        par[i].y = rnd01() * side;
        par[i].vx = (rnd01() - 0.5) * side / ncside / 5.0;
        par[i].vy = (rnd01() - 0.5) * side / ncside / 5.0;
        par[i].m = rnd01() * 0.01 * (ncside * ncside) / n_part / G * EPSILON2;
        par[i].active = 1;
    }
}

double compute_force(double m1, double m2, double dx, double dy);
void update_particles(Particle *particles, long long n_part, double side);

int main(int argc, char *argv[]) {
    if (argc != 6) {
        fprintf(stderr, "Uso: %s <semente> <lado> <grade> <num_particulas> <passos_tempo>\n", argv[0]);
        return EXIT_FAILURE;
    }

    long seed = atol(argv[1]);
    double side = atof(argv[2]);
    long ncside = atol(argv[3]);
    long long n_part = atoll(argv[4]);
    int steps = atoi(argv[5]);

    Particle *particles = (Particle *)malloc(n_part * sizeof(Particle));
    if (!particles) {
        fprintf(stderr, "Erro ao alocar memória para partículas!\n");
        return EXIT_FAILURE;
    }

    init_particles(seed, side, ncside, n_part, particles);

    double exec_time = -omp_get_wtime();
    for (int step = 0; step < steps; step++) {
        update_particles(particles, n_part, side);
    }
    exec_time += omp_get_wtime();

    printf("%.3f %.3f\n", particles[0].x, particles[0].y);
    printf("%d\n", 0);  // Número de colisões (ainda não implementado)
    
    fprintf(stderr, "Tempo de execução: %.1fs\n", exec_time);
    free(particles);
    return EXIT_SUCCESS;
}

void update_particles(Particle *particles, long long n_part, double side) {
    for (long long i = 0; i < n_part; i++) {
        if (!particles[i].active) continue;
        
        double fx = 0, fy = 0;
        for (long long j = 0; j < n_part; j++) {
            if (i == j || !particles[j].active) continue;
            
            double dx = particles[j].x - particles[i].x;
            double dy = particles[j].y - particles[i].y;
            double f = compute_force(particles[i].m, particles[j].m, dx, dy);
            
            fx += f * dx;
            fy += f * dy;
        }
        
        double ax = fx / particles[i].m;
        double ay = fy / particles[i].m;
        
        particles[i].vx += ax * DELTAT;
        particles[i].vy += ay * DELTAT;
        
        particles[i].x += particles[i].vx * DELTAT;
        particles[i].y += particles[i].vy * DELTAT;
        
        if (particles[i].x < 0) particles[i].x += side;
        if (particles[i].x > side) particles[i].x -= side;
        if (particles[i].y < 0) particles[i].y += side;
        if (particles[i].y > side) particles[i].y -= side;
    }
}

double compute_force(double m1, double m2, double dx, double dy) {
    double dist2 = dx * dx + dy * dy + EPSILON2;
    return G * m1 * m2 / (dist2 * sqrt(dist2));
}

