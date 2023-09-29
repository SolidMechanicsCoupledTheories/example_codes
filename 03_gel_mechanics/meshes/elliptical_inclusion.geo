//+
SetFactory("OpenCASCADE");


L0  = 50;
// Ellipse major axis
a = 2;
// Ellipse minor axis
b = 1;

lc = 0.1;
//+
Point(1) = {0, 0, 0, lc};
//+
Point(2) = {a, 0, 0, lc};
//+
Point(3) = {-a, 0, 0, lc};
//+
Point(4) = {0, -b, 0, lc};
//+
Point(5) = {0, b, 0, lc};
//*************************
Point(6) = {2*a, 2*b, 0, lc};
//+
Point(7) = {-2*a, 2*b, 0, lc};
//+
Point(8) = {-2*a, -2*b, 0, lc};
//+
Point(9) = {2*a, -2*b, 0, lc};

//+
Rectangle(10) = {-L0, -L0, 0, 2*L0, 2*L0, 0};
//+
Disk(11) = {0, 0, 0, a, b};

// We apply a boolean difference to  subtract Surface(7) from Surface(6) 
// to create the inclusion in the matrix:
//
BooleanFragments{Surface{10}; Delete;  }{ Surface{11}; Delete; }
//+


//+
Physical Curve("left", 13) = {2};
//+
Physical Curve("right", 14) = {3};
//+
Physical Curve("bottom", 15) = {1};
//+
Physical Curve("top", 16) = {4};
//+
Physical Curve("inclusion_curve", 17) = {5};
//+
Physical Surface("inclusion", 18) = {11};
//+
Physical Surface("matrix", 19) = {12};
