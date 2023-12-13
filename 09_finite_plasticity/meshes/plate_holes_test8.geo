// Gmsh project created on Sun Nov 26 12:57:28 2023
SetFactory("OpenCASCADE");


lc  = 0.5;
lcr = 0.01;
r = 0.4;

Point(1) = {0, 0, 0, lc};
Point(2) = {10, 0, 0, lc};
Point(3) = {10, 4, 0, lc};
Point(4) = {0, 4, 0, lc};



Point(5) = {2.5, 3, 0};
Point(8) = {3.5, 1, 0};
Point(11) = {7.5, 3, 0};
Point(14) = {6.5, 1, 0};
Point(17) = {5, 2, 0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

//+
Circle(5) = {2.5, 3, 0, 0.4, 0, 2*Pi};
//+
Circle(6) = {3.5, 1, 0, 0.4, 0, 2*Pi};
//+
Circle(7) = {7.5, 3, 0, 0.4, 0, 2*Pi};
//+
Circle(8) = {6.6, 1, 0, 0.4, 0, 2*Pi};
//+
Circle(9) = {5, 2, 0, 0.4, 0, 2*Pi};
//+
Curve Loop(1) = {4, 1, 2, 3};
//+
Curve Loop(2) = {5};
//+
Curve Loop(3) = {9};
//+
Curve Loop(4) = {6};
//+
Curve Loop(5) = {8};
//+
Curve Loop(6) = {7};
//+
Plane Surface(1) = {1, 2, 3, 4, 5, 6};
//+
Extrude {0, 0, 0.2} {
  Surface{1}; Layers {1}; 
}
//+
Physical Surface("Left", 28) = {2};
//+
Physical Surface("Right", 29) = {4};
//+
Physical Volume("Total_Volume", 30) = {1};
