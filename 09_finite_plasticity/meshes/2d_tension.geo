
// Gmsh project created on Sat Oct 28 11:07:56 2023
lc1= 0.3;
lc2=0.4;
lc3=0.8;
//+
Point(1) = {0, 0, 0, lc1};
//+
Point(2) = {4.9, 0, 0, lc1};
//+
Point(3) = {5, 10, 0, lc2};
//+
Point(4) = {0, 10, 0, lc2};
//+
Point(5) = {10, 15, 0, lc3};
//+
Point(7) = {10, 17, 0, lc3};
//+
Point(8) = {0, 17, 0, lc3};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {5, 7};
//+
Line(4) = {7, 8};
//+
Line(5) = {8, 1};
//+
Point(9) = {10, 10, 0, .5};
//+
Circle(6) = {5, 9, 3};
//+
Curve Loop(1) = {1, 2, -6, 3, 4, 5};
//+
Plane Surface(1) = {1};
//+
Physical Curve("left", 7) = {5};
//+
Physical Curve("bottom", 8) = {1};
//+
Physical Curve("top", 9) = {4};
//+
Physical Surface("specimen", 10) = {1};
