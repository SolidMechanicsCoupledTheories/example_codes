// Gmsh project created on Tue Aug 15 14:09:37 2023
//+
Point(1) = {4.9, 0, 0, 2};
//+
Point(2) = {5.0, 0, 0, 2};
//+
Point(3) = {5.1, 0, 0, 2};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Extrude {0, 0, 5} {
  Curve{1}; Curve{2}; 
}


// Start the helix construction, pi/2 at a time
Extrude {{0, 0, 2.5}, {0, 0, 1}, {0, 0, 0}, Pi/2} {
  Surface{10}; Surface{6}; 
}
//+
Extrude {{0, 0, 2.5}, {0, 0, 1}, {0, 0, 0}, Pi/2} {
  Surface{32}; Surface{54}; 
}
//+
Extrude {{0, 0, 2.5}, {0, 0, 1}, {0, 0, 0}, Pi/2} {
  Surface{76}; Surface{98}; 
}
// 2*pi radians
Extrude {{0, 0, 2.5}, {0, 0, 1}, {0, 0, 0}, Pi/2} {
  Surface{120}; Surface{142}; 
}


//+
Extrude {{0, 0, 2.5}, {0, 0, 1}, {0, 0, 0}, Pi/2} {
  Surface{164}; Surface{186}; 
}
//+
Extrude {{0, 0, 2.5}, {0, 0, 1}, {0, 0, 0}, Pi/2} {
  Surface{208}; Surface{230}; 
}
//+
Extrude {{0, 0, 2.5}, {0, 0, 1}, {0, 0, 0}, Pi/2} {
  Surface{252}; Surface{274}; 
}
// 4*pi radians
Extrude {{0, 0, 2.5}, {0, 0, 1}, {0, 0, 0}, Pi/2} {
  Surface{296}; Surface{318}; 
}


//+
Extrude {{0, 0, 2.5}, {0, 0, 1}, {0, 0, 0}, Pi/2} {
  Surface{340}; Surface{362}; 
}
//+
Extrude {{0, 0, 2.5}, {0, 0, 1}, {0, 0, 0}, Pi/2} {
  Surface{384}; Surface{406}; 
}
//+
Extrude {{0, 0, 2.5}, {0, 0, 1}, {0, 0, 0}, Pi/2} {
  Surface{428}; Surface{450}; 
}
// 6*pi radians
Extrude {{0, 0, 2.5}, {0, 0, 1}, {0, 0, 0}, Pi/2} {
  Surface{472}; Surface{494}; 
}


// Base set - 539
Physical Surface(539) = {10, 6};
// Outside set - 540
Physical Surface(540) = {23, 67, 111, 155, 199, 243, 287, 331, 375, 419, 463, 507};
// Inside set - 541
Physical Surface(541) = {53, 97, 141, 185, 229, 273, 317, 361, 405, 449, 493, 537};
// Volume set - 542
Physical Volume(542) = {2, 1, 7, 8, 4, 3, 5, 6, 13, 14, 11, 12, 16, 15, 9, 10, 21, 22, 19, 20, 24, 23, 18, 17};
