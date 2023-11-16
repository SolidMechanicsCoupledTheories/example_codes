// Gmsh project created on Fri Oct 27 09:55:10 2023
SetFactory("OpenCASCADE");
res=2.5;
//+
Point(1) = {0, 0, 0, res};
//+
Point(2) = {43.30, 25, 0, res};
//+
Point(3) = {40.58, 29.21, 0, res};
//+
Point(4) = {9.22, 15.97, 0, res};
//+
Point(5) = {5, 49.74, 0, res};
//+
Point(6) = {0, 50, 0, res};
//+
Point(7) = {18.45, 0, 0, res};
//+
Point(8) = {0, 10, 0, res};
//+
Point(9) = {45.58, 20.54, 0, res};

//+
Line(1) = {6, 8};
//+
Line(2) = {8, 1};
//+
Line(3) = {1, 7};
//+
Bezier(4) = {2, 9, 7};
//+
Bezier(5) = {6, 5, 4};
//+
Bezier(6) = {4, 3, 2};
//+
Curve Loop(1) = {5, 6, -4, -3, -2, -1};
//+
Plane Surface(1) = {1};
//+
Extrude {0, 0, 0.5} {
  Surface{1}; Layers{2}; 
}
//+
Extrude {0, 0, -0.5} {
  Surface{1}; Layers{2}; 
}
//+
Physical Volume(71) = {1, 2};
//+
Physical Surface("Ztop", 72) = {8};
//+
Physical Surface("Zmid", 73) = {1};
//+
Physical Surface("Zbot", 74) = {15};
//+
Physical Surface("Xbot", 75) = {6,7,13,14};
//+
Physical Surface("Ybot", 76) = {5,12};
