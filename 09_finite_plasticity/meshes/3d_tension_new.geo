
// Gmsh project created on Sat Oct 28 11:07:56 2023
lc1= 0.4;
lc2=0.5;
lc3=1.0;
r  =2;
//+
Point(1) = {0, 0, 0, lc1};
//+
Point(2) = {4.9, 0, 0, lc1};
//+
Point(3) = {5, 5, 0, lc1};
//+
Point(4) = {5+r, 5, 0};
//+
Point(5) = {5+r, 5+r, 0, lc3};
//+
Point(6) = {5+r, 5+5*r, 0, lc3};
//+
Point(7) = {0, 5+5*r, 0, lc3};
 
//+
Line(1) = {5, 6};
//+
Line(2) = {6, 7};
//+
Line(3) = {7, 1};
//+
Line(4) = {1, 2};
//+
Line(5) = {2, 3};
//+
Circle(6) = {5, 4, 3};
//+
Curve Loop(1) = {4, 5, -6, 1, 2, 3};
//+
Plane Surface(1) = {1};
//+
Extrude {0, 0, 0.5} {
  Surface{1}; Layers {1}; 
}
//+
Physical Surface("left", 39) = {37};
//+
Physical Surface("bot", 40) = {17};
//+
Physical Surface("top", 41) = {33};
//+
Physical Surface("right", 42) = {29};
//+
Physical Volume("total_vol", 43) = {1};

