// Gmsh project created on Sun Nov 26 12:57:28 2023
SetFactory("OpenCASCADE");

Point(1) = {0, 0, 0, 5.0};
Point(2) = {100, 0, 0, 5.0};
Point(3) = {100, 50, 0, 5.0};
Point(4) = {0, 50, 0, 5.0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Circle(5) = {25, 35, 0, 5.0, 0, 2*Pi};
Circle(6) = {75, 35, 0, 5.0, 0, 2*Pi};
//
Circle(7) = {35, 25, 0, 5.0, 0, 2*Pi};
Circle(8) = {65, 25, 0, 5.0, 0, 2*Pi};
Circle(9) = {50, 15, 0, 5.0, 0, 2*Pi};

//+
Curve Loop(1) = {4, 1, 2, 3};
//+
Curve Loop(2) = {5};
//+
Curve Loop(3) = {7};
//+
Curve Loop(4) = {9};
//+
Curve Loop(5) = {8};
//+
Curve Loop(6) = {6};
//+
Plane Surface(1) = {1, 2, 3, 4, 5, 6};
//+
Extrude {0, 0, 5} {
  Surface{1}; Layers {1}; 
}
//+
Physical Surface("left", 28) = {2};
//+
Physical Surface("right", 29) = {4};
//+
Physical Volume("total", 30) = {1};
//+
Physical Volume("total", 30) += {1};
