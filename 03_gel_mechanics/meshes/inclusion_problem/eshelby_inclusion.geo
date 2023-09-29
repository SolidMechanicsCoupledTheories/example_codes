//+
SetFactory("OpenCASCADE");

//+
Rectangle(1) = {-7, -7, 0, 14, 14, 0};
//+
Disk(2) = {0, 0, 0, 1, 1};

// We apply a boolean difference to  subtract Surface(2) from Surface(1) 
// to create the inclusion in the matrix:
//
BooleanFragments{Surface{1}; Delete;  }{ Surface{2}; Delete; }
//+


