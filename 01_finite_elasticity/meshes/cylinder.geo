//+
SetFactory("OpenCASCADE");
L = 25.4;
R = 12.7;
//+
Cylinder(1) = {0, 0, 0, L, 0, 0, R, 2*Pi};
//+
Physical Surface("xBot", 101) = {3};
//+
Physical Surface("xTop", 102) = {2};
//+
Physical Surface("wall", 7) = {1};
//+
Physical Volume("Tot_Vol", 6) = {1};

