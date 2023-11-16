// Gmsh project created on Wed Aug 16 15:42:01 2023
//+
Point(1) = {-2.5, -0.1, 0, 1};
//+
Point(2) = {-2.5, 0, 0, 1};
//+
Point(3) = {-2.5, 0.1, 0, 1};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Extrude {5, 0, 0} {
  Curve{1}; Curve{2}; 
}
//+
Extrude {{0, 0, 50}, {0, 0, 1}, {0, 0, 0}, 2*Pi }{
  Surface{10}; Surface{6}; 
}//+
Physical Volume(55) = {1, 2};
// Outside surface - 56
Physical Surface(56) = {23};
// Inside surface - 57
Physical Surface(57) = {53};
// Base - 58
Physical Surface(58) = {10, 6};
