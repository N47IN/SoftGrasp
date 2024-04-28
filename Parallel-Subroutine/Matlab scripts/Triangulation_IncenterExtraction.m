% R Navin Sriram, n47insriram@gmail.com
%
% Importing and creating a mesh from a STL file object

model = createpde;
impGeo = importGeometry(model,'mesh.stl');
mesh= generateMesh(model,"Hmin",50);
[p,edges,t]=meshToPet(mesh);

% % Through Convex Hull
% 
% K1 = convhull(p(1,:),p(2,:),p(3,:));
% subplot(1,2,1)
% defaultFaceColor  = [0.6875 0.8750 0.8984];
% trisurf(K1,p(1,:),p(2,:),p(3,:),'Facecolor',defaultFaceColor)
% axis equal
 

% Delaunay triangulation for triangulating the outer surface

DT = delaunayTriangulation(transpose(p(1,:)),transpose(p(2,:)),transpose(p(3,:)));
[T,Xb] = freeBoundary(DT);
TR = triangulation(T,Xb); 
G_pts = incenter(TR);
G_norm = faceNormal(TR);  
% trisurf(T,Xb(:,1),Xb(:,2),Xb(:,3), ...
%      'FaceColor','cyan','FaceAlpha',0.8);
axis equal
hold on  
scatter3(G_pts(:,1),G_pts(:,2),G_pts(:,3),10,'filled','r');



reach(G_pts)

function dist=distance(x,k,j)
 dist=sqrt((x(k,1)-x(j,1)).^2 + (x(k,2)-x(j,2)).^2 + (x(k,3)-x(j,3)).^2);
end



function shortlist1=reach(p)
i=0;
k=200;
[row,col]=size(p);
shortlist1=[];
 for j =1:col
     if distance(p,j,k)<3000

        i=i+1;
        hold on
         scatter3(p(1,j),p(2,j),p(3,j),20,'filled','b')
        hold off
        i
     end  
 end
end
