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

Point(8) = {3.5, 1, 0};
Point(9) = {3.5+r, 1, 0, lcr};
Point(10) = {3.5-r, 1, 0, lcr};

Point(11) = {7.5, 3, 0};
Point(12) = {7.5+r, 3, 0, lcr};
Point(13) = {7.5-r, 3, 0, lcr};

Point(14) = {6.5, 1, 0};
Point(15) = {6.5+r, 1, 0, lcr};
Point(16) = {6.5-r, 1, 0, lcr};

Point(17) = {5, 2, 0};
Point(18) = {5+r, 2, 0, lcr};
Point(19) = {5-r, 2, 0, lcr};





Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};


 

//+
Circle(5) = {6, 5, 7};
//+
Circle(6) = {7, 5, 6};


//+
Circle(7) = {9, 8, 10};
//+
Circle(8) = {10, 8, 9};
//+
Circle(9) = {18, 17, 19};
//+
Circle(10) = {19, 17, 18};
//+
Circle(11) = {12, 11, 13};
//+
Circle(12) = {13, 11, 12};
//+
Circle(13) = {15, 14, 16};
//+
Circle(14) = {16, 14, 15};
//+
Curve Loop(1) = {4, 1, 2, 3};
//+
Curve Loop(2) = {6, 5};
//+
Curve Loop(3) = {7, 8};
//+
Curve Loop(4) = {10, 9};
//+
Curve Loop(5) = {12, 11};
//+
Curve Loop(6) = {13, 14};
//+
Plane Surface(1) = {1, 2, 3, 4, 5, 6};
//+
Physical Curve("left", 15) = {4};
//+
Physical Curve("right", 16) = {2};
//+
Physical Surface("total_surf", 17) = {1};
