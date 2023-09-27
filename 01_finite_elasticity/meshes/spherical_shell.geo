// Gmsh project created on Fri Sep  1 10:47:23 2023
SetFactory("OpenCASCADE");

Ro  = 11;
Ri = 10;
//+
Sphere(1) = {0, 0, 0, Ro, 0, Pi/2, Pi/2};
//+
Sphere(2) = {0, 0, 0, Ri, 0, Pi/2, Pi/2};

// We apply a boolean difference to  subtract Volume(2) from Volume(1) 
// to create a thick-walled sphere:
//
BooleanDifference(3) = { Volume{1}; Delete; }{ Volume{2}; Delete; };

Characteristic Length{ PointsOf{ Volume{3}; } } = 0.75;
//+
Physical Surface("yBot", 12) = {4};
//+
Physical Surface("zBot", 13) = {3};
//+
Physical Surface("xBot", 14) = {2};

//+
Physical Surface("inner_surf", 15) = {5};
//+
Physical Volume("total_volume", 16) = {3};
