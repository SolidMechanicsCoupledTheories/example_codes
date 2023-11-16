// Gmsh project created on Tue Aug 15 11:46:10 2023
//+
Point(1) = {19.75, 0, 0, 2.5};
//+
Point(2) = {20.0, 0, 0, 2.5};
//+
Point(3) = {20.25, 0, 0, 2.5};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Extrude {0, 0, 100} {
  Curve{1}; Curve{2}; 
}
//+
Extrude {{0, 0, 1}, {0, 0, 0}, Pi/2} {
  Surface{10}; Surface{6};
}
// Top
Physical Surface(55) = {19, 41};
// Bottom
Physical Surface(56) = {27, 49};
// Outside
Physical Surface(57) = {23};
// Inside
Physical Surface(58) = {53};
// Volume
Physical Volume(59) = {1, 2};
// xNormal
Physical Surface(60) = {32, 54};
// yNormal
Physical Surface(61) = {32, 10, 6};
// xLine
Physical Curve(62) = {4};
