//+
Point(1) = {0, 0, 0, 0.025};
//+
Point(2) = {2, 0, 0, 0.025};
//+
Point(3) = {2, 0.5, 0, 0.025};
//+
Point(4) = {0, 0.5, 0, 0.025};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 1};
//+
Curve Loop(1) = {1, 2, 3, 4};
//+
Plane Surface(1) = {1};
//+
Symmetry {0, 1, 0, 0} {
  Duplicata { Surface{1}; }
}
//+
Physical Surface("domain", 13) = {1, 5};
//+
Physical Curve("clamp", 14) = {4, 9};
//+
Physical Point("load", 15) = {2};
