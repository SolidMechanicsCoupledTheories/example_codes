// Gmsh project created on Sun Nov 26 12:57:28 2023
SetFactory("OpenCASCADE");

// Let's create a simple rectangular geometry
lc = .15;
Point(1) = {0.0,0.0,0,lc}; Point(2) = {1,0.0,0,lc};
Point(3) = {1,1,0,lc};     Point(4) = {0,1,0,lc};
Point(5) = {0.5,.5,0,lc};

Line(1) = {1,2}; Line(2) = {2,3}; Line(3) = {3,4}; Line(4) = {4,1};

Curve Loop(5) = {1,2,3,4}; 
//+
Circle(5) = {0.5, 0.5, 0, 0.2, 0, 2*Pi};
//+
Curve Loop(6) = {4, 1, 2, 3};
//+
Plane Surface(1) = {6};
//+
Curve Loop(7) = {5};
//+
Plane Surface(2) = {7};
//+
BooleanIntersection(8) = { Volume{2}; Delete; }{ Volume{1}; Delete; };

//+
Curve Loop(8) = {4, 1, 2, 3};
//+
Curve Loop(9) = {5};
//+
Plane Surface(3) = {8, 9};
//+
Physical Curve("left", 10) = {4};
//+
Physical Curve("right", 11) = {2};
//+
Physical Curve("hole_surf", 12) = {5};
