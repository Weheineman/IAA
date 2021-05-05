#include <stdlib.h>
#include <stdio.h>

// Auxiliar function to get a double random in a range, since
// rand() returns an int in [0, RAND_MAX].
double random_from_range(double rangeMin, double rangeMax)
{
  double range = (rangeMax - rangeMin);
  double div = RAND_MAX / range;
  return rangeMin + (rand() / div);
}

typedef struct point
{
  double *coordinates;
  int dimension;
} point;

point *point_new(int dimension)
{
  point *p = (point *)malloc(sizeof(point));
  p->dimension = dimension;
  p->coordinates = (double *)malloc(dimension * sizeof(double));
  return p;
}

point *point_new_init(int dimension, double *coordinates)
{
  point *p = point_new(dimension);
  for (int i = 0; i < dimension; ++i)
    p->coordinates[i] = coordinates[i];
  return p;
}

void point_delete(point *p)
{
  free(p->coordinates);
  free(p);
}

// Generates a random point where each coordinate
// differs from the center at most rangeDelta.
point *point_new_random(point *center, double rangeDelta)
{
  point *p = point_new(center->dimension);
  for (int i = 0; i < center->dimension; ++i)
    p->coordinates[i] = random_from_range(center->coordinates[i] - rangeDelta,
                                          center->coordinates[i] + rangeDelta);
  return p;
}

void point_fprint(point *p, FILE *outputFile, int class)
{
  fprintf(outputFile, "%lf", p->coordinates[0]);
  for (int i = 1; i < p->dimension; ++i)
    fprintf(outputFile, ", %lf", p->coordinates[i]);
  fprintf(outputFile, ", %d.\n", class);
}