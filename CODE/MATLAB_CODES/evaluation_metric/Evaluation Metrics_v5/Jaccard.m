function jaccard = Jaccard(S,G)
S = single(S);
G = single(G);
inter_image = S&G;
union_image = S|G;
jaccard = sum(inter_image(:))/sum(union_image(:));
end