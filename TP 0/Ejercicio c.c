#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

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
  double x, y;
} point;

point point_new_random()
{
  point p;
  p.x = random_from_range(-1, 1);
  p.y = random_from_range(-1, 1);
  return p;
}

double point_r(point p)
{
  return sqrt(p.x * p.x + p.y * p.y);
}

double point_theta(point p)
{
  return atan2(p.y, p.x);
}

void point_fprint(point p, FILE *outputFile, int class)
{
  fprintf(outputFile, "%lf, %lf, %d.\n", p.x, p.y, class);
}

int is_in_circle(point p)
{
  return point_r(p) <= 1;
}

int is_in_class_0(point p)
{
  double theta = point_theta(p);
  double r = point_r(p);
  double pi = acos(-1);
  double curve = theta / (4 * pi);
  return ((r >= curve && r <= curve + 0.25) ||
          (r >= curve + 0.5 && r <= curve + 0.75) ||
          (r >= curve + 1));
}

int is_in_class_1(point p)
{
  return !is_in_class_0(p);
}

void write_names_file(char *namesFileName)
{
  FILE *namesFile = fopen(namesFileName, "w");

  fprintf(namesFile, "0, 1.\n");
  for (int i = 0; i < 2; ++i)
    fprintf(namesFile, "coord%i: continuous.\n", i);

  fclose(namesFile);
}

int main(int argc, char *argv[])
{
  if (argc != 3)
  {
    printf("Argument error. Usage: executable fileName n\n");
    return 1;
  }

  int n = atoi(argv[2]);
  char *fileName = argv[1];

  char dataFileName[64], namesFileName[64];
  sprintf(dataFileName, "%s.data", fileName);
  sprintf(namesFileName, "%s.names", fileName);
  write_names_file(namesFileName);
  FILE *dataFile = fopen(dataFileName, "w");

  // Get random seed from time.
  srand(time(NULL));

  // Class 0
  for (int generatedPoints = 0; generatedPoints < (n + 1) / 2;)
  {
    point p = point_new_random();
    if (is_in_circle(p) && is_in_class_0(p))
    {
      point_fprint(p, dataFile, 0);
      generatedPoints++;
    }
  }

  // Class 1
  for (int generatedPoints = 0; generatedPoints < n / 2;)
  {
    point p = point_new_random();
    if (is_in_circle(p) && is_in_class_1(p))
    {
      point_fprint(p, dataFile, 1);
      generatedPoints++;
    }
  }

  fclose(dataFile);
  return 0;
}