// Gmsh project created on Fri Oct 27 09:55:10 2023
res=2.5;
//+
Point(1) = {0, 0, 0, res};
//+
Point(2) = {50, 0, 0, res};
//+
Point(3) = {50, 10, 0, res};
//+
Point(4) = {10, 10, 0, res};
//+
Point(5) = {10, 50, 0, res};
//+
Point(6) = {0, 50, 0, res};

//+
Point(7) = {10, 0, 0, res};
//+
Point(8) = {0, 10, 0, res};
//+
Line(1) = {1, 7};
//+
Line(2) = {7, 2};
//+
Line(3) = {2, 3};
//+
Line(4) = {3, 4};
//+
Line(5) = {7, 4};
//+
Line(6) = {4, 5};
//+
Line(7) = {5, 6};
//+
Line(8) = {6, 8};
//+
Line(9) = {8, 1};
//+
Line(10) = {4, 8};
//+
Curve Loop(1) = {2, 3, 4, -5};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {1, 5, 10, 9};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {8, -10, 6, 7};
//+
Plane Surface(3) = {3};

//+
Extrude {0, 0, 0.5} {
  Surface{1}; Surface{2}; Surface{3}; Layers {2};
}

//+
Extrude {0, 0, -0.5} {
  Surface{1}; Surface{2}; Surface{3}; Layers {2};
}


//+
Physical Surface("Zmid", 31) = {1, 2, 3};
//+
Physical Surface("Ztop", 32) = {32, 54, 76};
//+
Physical Surface("Zbot", 33) = {98, 120, 142};
//+
Physical Surface("Xbot", 34) = {53, 119, 129, 63};
//+
Physical Surface("Ybot", 35) = {107, 41, 19, 85};

//+
Physical Volume(78) = {4, 1, 5, 2, 6, 3};
