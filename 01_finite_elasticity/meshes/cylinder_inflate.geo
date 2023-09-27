// Gmsh project created on Tue Aug 15 11:46:10 2023
//+
Geometry.PointNumbers = 1;
Geometry.Color.Points = Red;
General.Color.Text = White;
Mesh.Color.Points = Blue;
//+
Point(1) = {10, 0, 0, 0.5};
//+
Point(2) = {11, 0, 0, 0.5};
//+
Line(1) = {1, 2};
//+
Extrude {0, 0, 5} {
  Curve{1}; Layers {5}; 
}
//+
Curve Loop(1) = {3, 2, -4, -1};
//+
Plane Surface(6) = {1};

//+
Extrude {{0, 0, 1}, {0, 0, 0}, Pi/2} {
  Surface{5}; Layers{10}; 
}
//+
Physical Surface("right_bot", 29) = {5};
//+
Physical Surface("left_top", 30) = {28};
//+
Physical Surface("inner_surf", 31) = {27};
//+
Physical Surface("z_bot", 32) = {15};
//+
Physical Surface("z_top", 33) = {23};
//+
Physical Volume("volume", 34) = {1};
