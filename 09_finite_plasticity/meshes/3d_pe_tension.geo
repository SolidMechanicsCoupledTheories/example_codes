
// Gmsh project created on Sat Oct 28 11:07:56 2023
lc1= 0.1;
lc2=0.2;
lc3=0.5;
r  =0.5;
//+
Point(1) = {0, 0, 0, lc1};
//+
Point(2) = {0.495, 0, 0, lc1};
//+
Point(3) = {0.5, 1.0, 0, lc1};
//+
Point(4) = {0.5+r, 1.0, 0};
//+
Point(5) = {0.5+r, 1.0+r, 0, lc3};
//+
Point(6) = {0.5+r, 1.0+4*r, 0, lc3};
//+
Point(7) = {0, 1.0+4*r, 0, lc3};
 
//+
Point(8) = {-(0.5+r), 1.0+4*r, 0, lc3};
//+
Point(9) = {-(0.5+r), 1.0+r, 0, lc3};
//+
Point(10) = {-(0.5+r), 1.0, 0};
//+
Point(11) = {-0.5, 1.0, 0, lc1};

//+
Point(12) = {-0.495, 0, 0, lc1};//+

//+
Line(1) = {11, 12};
//+
Line(2) = {12, 1};
//+
Line(3) = {1, 2};
//+
Line(4) = {2, 3};
//+
Circle(5) = {5, 4, 3};
//+
Line(6) = {5, 6};
//+
Line(7) = {6, 7};
//+
Line(8) = {7, 8};
//+
Line(9) = {8, 9};
//+
Circle(10) = {11, 10, 9};
//+
Line(11) = {5, 9};
//+
Curve Loop(1) = {3, 4, -5, 11, -10, 1, 2};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {9, -11, 6, 7, 8};
//+
Plane Surface(2) = {2};
//+
Extrude {0, 0, 5} {
  Surface{2}; Surface{1}; Layers {10}; 
}
//+
Physical Surface("bottom", 76) = {50, 74};
//+
Physical Surface("top", 77) = {33, 37};
//+
Physical Surface("left", 78) = {2};
//+
Physical Surface("right", 79) = {38};
//+
Physical Volume("total_volume", 80) = {1, 2};
