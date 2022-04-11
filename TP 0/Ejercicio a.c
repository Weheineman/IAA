#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "point.h"

double pointProbabilityDensity(point *x, point *center, double standardDeviation)
{
  double ans = 1;
  double pi = acos(-1);
  double denominator = sqrt(2 * pi) * standardDeviation;
  for (int i = 0; i < x->dimension; ++i)
  {
    double exponent = -0.5 / pow(standardDeviation, 2) * pow(x->coordinates[i] - center->coordinates[i], 2);
    ans *= 1.0 / denominator * exp(exponent);
  }
  return ans;
}

double probabilityDensity(double xValue, double standardDeviation){
  double pi = acos(-1);
  double denominator = sqrt(2 * pi) * standardDeviation;
  double exponent = -0.5 / pow(standardDeviation, 2) * pow(x->coordinates[i] - center->coordinates[i], 2);
  return denominator * exp(exponent);
}

double generateCoordinate(double center, double standardDeviation){
  double rangeDelta = 5 * standardDeviation;
  double coordinate = random_from_range(center - rangeDelta, center + rangeDelta);
  double maxY = probabilityDensity(center, center, standardDeviation);
}

void generate_points(FILE *dataFile, int amount, double C, point *center, int class)
{
  double standardDeviation = C * sqrt(center->dimension);
  double rangeDelta = 5 * standardDeviation;
  double maxY = probabilityDensity(center, center, standardDeviation);

  for (int generatedPoints = 0; generatedPoints < amount;)
  {
    point *x = point_new_random(center, rangeDelta);
    double y = random_from_range(0, maxY);
    if (y < probabilityDensity(x, center, standardDeviation))
    {
      generatedPoints++;
      point_fprint(x, dataFile, class);
    }
    point_delete(x);
  }
}

void write_names_file(char *namesFileName, int dimension)
{
  FILE *namesFile = fopen(namesFileName, "w");

  fprintf(namesFile, "0, 1.\n");
  for (int i = 0; i < dimension; i++)
    fprintf(namesFile, "coord%d: continuous.\n", i);

  fclose(namesFile);
}

int main(int argc, char *argv[])
{
  if (argc != 5)
  {
    printf("Argument error. Usage: executable fileName d n C\n");
    return 1;
  }

  int n = atoi(argv[3]), d = atoi(argv[2]);
  double C = atof(argv[4]);
  char *fileName = argv[1];

  char dataFileName[64], namesFileName[64];
  sprintf(dataFileName, "%s.data", fileName);
  sprintf(namesFileName, "%s.names", fileName);
  write_names_file(namesFileName, d);
  FILE *dataFile = fopen(dataFileName, "w");

  // Get random seed from time.
  srand(time(NULL));

  // Class 0
  double class0Coords[d];
  for (int i = 0; i < d; ++i)
    class0Coords[i] = 1;
  point *class0Center = point_new_init(d, class0Coords);
  int class0Size = (n + 1) / 2;
  generate_points(dataFile, class0Size, C, class0Center, 0);
  point_delete(class0Center);

  // Class 1
  double class1Coords[d];
  for (int i = 0; i < d; ++i)
    class1Coords[i] = -1;
  point *class1Center = point_new_init(d, class1Coords);
  int class1Size = n - class0Size;
  generate_points(dataFile, class1Size, C, class1Center, 1);
  point_delete(class1Center);

  fclose(dataFile);
  return 0;
}
