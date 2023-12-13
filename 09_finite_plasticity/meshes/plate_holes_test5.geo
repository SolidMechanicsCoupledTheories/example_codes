// Gmsh project created on Sun Nov 26 12:57:28 2023


lc  = 1;
lcr = 0.01;
r = 0.4;

Point(1) = {0, 0, 0, lc};
Point(2) = {10, 0, 0, lc};
Point(3) = {10, 4, 0, lc};
Point(4) = {0, 4, 0, lc};

Point(5) = {2.5, 3, 0};
Point(6) = {2.5+r, 3, 0, lcr};
Point(7) = {2.5-r, 3, 0, lcr};

Point(8) = {7.5, 3, 0};
Point(9) = {7.5+r, 3, 0, lcr};
Point(10) = {7.5-r, 3, 0, lcr};

Point(11) = {5, 2, 0};
Point(12) = {5+r, 2, 0, lcr};
Point(13) = {5-r, 2, 0, lcr};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};


 

//+
Circle(5) = {6, 5, 7};
//+
Circle(6) = {7, 5, 6};
//+
Circle(7) = {12, 11, 13};
//+
Circle(8) = {13, 11, 12};
//+
Circle(9) = {9, 8, 10};
//+
Circle(10) = {10, 8, 9};
//+
Curve Loop(1) = {4, 1, 2, 3};
//+
Curve Loop(2) = {6, 5};
//+
Curve Loop(3) = {8, 7};
//+
Curve Loop(4) = {10, 9};
//+
Plane Surface(1) = {1, 2, 3, 4};
//+
Physical Curve("left", 11) = {4};
//+
Physical Curve("right", 12) = {2};
//+
Physical Surface("total_surf", 13) = {1};

