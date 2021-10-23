function [g, gp] = ReLU(x)
if x<0
    g=0;
    gp=0;
else
    g=x;
    gp=0;
end
end