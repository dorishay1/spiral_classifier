function [g ,gp] = ReLU(x)
[a,b] = size(x);
sz = a*b;
g= zeros(size(x));
gp= zeros(size(x));
for i= 1:sz
    g(i) = max(0,x(i));
end
for i = 1:sz
    if g(i) == 0
        gp(i) = 0;
    else
        gp(i)=1;
    end
end
end