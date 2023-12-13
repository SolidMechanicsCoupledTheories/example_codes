
// Gmsh project created on Sat Oct 28 11:07:56 2023
lc1= 0.075;
lc2=0.2;
lc3=0.3;
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
Point(5) = {0.5+r, 1.0+r, 0, lc2};
//+
Point(6) = {0.5+r, 1.0+3*r, 0, lc2};
//+
Point(7) = {0, 1.0+3*r, 0, lc3};
//+
Point(8) = {0, 1.0, 0, lc1};
 


//+
Line(1) = {5, 6};
//+
Line(2) = {6, 7};
//+
Line(3) = {7, 8};
//+
Line(4) = {8, 1};
//+
Line(5) = {1, 2};
//+
Line(6) = {2, 3};

//+
Circle(7) = {5, 4, 3};
//+
Line(8) = {3, 8};
//+
Curve Loop(1) = {5, 6, 8, 4};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {7, 8, -3, -2, -1};
//+
Plane Surface(2) = {2};


//+
Extrude {{0, 1, 0}, {0, 0, 0}, Pi/2} {
  Surface{2}; Surface{1}; 
}

//+
Physical Surface("ybot", 48) = {38};
//+
Physical Surface("ytop", 49) = {25};
//+
Physical Surface("zbot", 50) = {1, 2};
//+
Physical Surface("xbot", 51) = {47, 30};
//+
Physical Volume("total_volume", 52) = {1, 2};
