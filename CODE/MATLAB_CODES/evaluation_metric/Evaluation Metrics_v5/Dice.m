function dice = Dice(S,G)
S = single(S);
G = single(G);
temp = S&G;
dice = 2*sum(temp(:))/(sum(S(:))+sum(G(:)));
end