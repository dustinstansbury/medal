function [fx,dfx]=gradSin(X);

fx = sin(X);
dfx = -cos(X);