// Gmsh project created on Tue Aug 15 11:46:10 2023
//+
Geometry.PointNumbers = 1;
Geometry.Color.Points = Red;
General.Color.Text = White;
Mesh.Color.Points = Blue;
//+
lc =0.008;
//+
Point(1) = {0, 0, 0, lc};
//+
Point(2) = {0.5, 0, 0, lc};
//+
Point(3) = {0, 0.5, 0, lc};
//+
Line(1) = {1, 2};
//+
Line(2) = {1, 2};
//+
Line(3) = {1, 3};

//+
Circle(4) = {2, 1, 3};
//+
Curve Loop(1) = {3, -4, -1};
//+
Plane Surface(1) = {1};
//+
Physical Curve("left", 5) = {3};
//+
Physical Curve("bottom", 6) = {1};
//+
Physical Curve("outer", 7) = {4};
//+
Physical Surface("domain", 8) = {1};
