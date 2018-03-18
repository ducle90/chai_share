function d = point_to_line(pt, v1, v2)


% d = abs( det([pt-v1;v2-v1]) )/norm(v2-v1) ; % this only works for 2-d
% space

a = v1 - v2;

b = pt - v2;
if (norm(a)==0 || norm(b) ==0 || norm(pt-v1)==0)
    d=0;
else
    d = norm(b) * sqrt(1- (dot(a,b)/(norm(a)*norm(b)))^2);  % |b|sin(theta)
end
end