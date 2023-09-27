// Gmsh project created on Mon Aug 28 11:09:08 2023
//+
Point(1) = {3, 0, 0, 1};
//+
Point(2) = {15, 0, 0, 1};
//+
Point(3) = {15, 10, 0, 1};
//+
Point(4) = {0, 10, 0, 1};
//+
Point(5) = {0, 3, 0, 1};
//+
Point(6) = {0, 0, 0, 1};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 5};
//+
Circle(5) = {5, 6, 1};
//+
Curve Loop(1) = {1, 2, 3, 4, 5};
//+
Plane Surface(1) = {1};
//+
Extrude {0, 0, 1} {
  Surface{1}; Layers {1}; 
}
//+
Physical Surface("xbot", 33) = {27};
//+
Physical Surface("ybot", 34) = {15};
//+
Physical Surface("xtop", 35) = {19};
//+
Physical Volume("Specimen", 36) = {1};
