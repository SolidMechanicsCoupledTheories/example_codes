
// Gmsh project created on Sat Oct 28 11:07:56 2023
lc1= 0.65;
lc2=0.65;
lc3=0.65;
//+
Point(1) = {0, 0, 0, lc1};
//+
Point(2) = {4.9, 0, 0, lc1};
//+
Point(3) = {5, 10, 0, lc1};
//+
Point(4) = {0, 10, 0, lc1};
//+
Point(5) = {10, 15, 0, lc2};
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
Point(9) = {10, 10, 0, lc3};
//+
Circle(6) = {5, 9, 3};
//+
Curve Loop(1) = {1, 2, -6, 3, 4, 5};
//+
Plane Surface(1) = {1};

//+
Plane Surface(2) = {1};
//+
Extrude {0, 0, 1.0} {
  Surface{1}; Layers {1}; 
}

//+
Physical Surface("left", 39) = {37};
//+
Physical Surface("bottom", 40) = {17};
//+
Physical Surface("top", 41) = {33};
//+
Physical Surface("zbot", 43) = {1};
//+
Physical Volume("specimen", 42) = {1};

