// Gmsh project created on Fri Sep  8 20:13:09 2023
//+
Point(1) = {0, 0, 0, 0.1};
//+
Point(2) = {1,0, 0, 0.1};
//+
Point(3) = {0, 1, 0, 0.1};
//+
Circle(1) = {2, 1, 3};
//+
Line(2) = {1, 2};
//+
Line(3) = {1, 3};
//+
Curve Loop(1) = {2, 1, -3};
//+
Plane Surface(1) = {1};
//+
Physical Curve("bottom", 4) = {2};
//+
Physical Curve("left", 5) = {3};
//+
Physical Curve("arc", 6) = {1};
//+
Physical Surface("plane", 7) = {1};
