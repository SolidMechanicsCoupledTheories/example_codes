// Gmsh project created on Sun Nov 26 12:57:28 2023


lc1= 5;
lc2= 1.0;

Point(1) = {0, 0, 0, lc};
Point(2) = {10, 0, 0, lc};
Point(3) = {10, 5, 0, lc};
Point(4) = {0, 5, 0, lc};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};



//+
Point(5) = {3, 3, 0, 1.0, lc2};
//+
Point(6) = {3.5, 3, 0, lc2};
//+
Point(7) = {2.5, 3, 0, lc2};

//+
Circle(5) = {6, 5, 7};
//+
Circle(6) = {7, 5, 6};
//+
Curve Loop(1) = {4, 1, 2, 3};
//+
Curve Loop(2) = {6, 5};
//+
Plane Surface(1) = {1, 2};
//+
Physical Curve("left", 7) = {4};
//+
Physical Curve("right", 8) = {2};
//+
Physical Surface("surf_tot", 9) = {1};
