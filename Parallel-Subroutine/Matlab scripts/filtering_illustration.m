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


k=200;

reach(G_pts,k)

function dist=distance(x,k,j)
 dist=sqrt((x(k,1)-x(j,1)).^2 + (x(k,2)-x(j,2)).^2 + (x(k,3)-x(j,3)).^2);
end


% function reach_sphere(p,j)
% % Generate the x, y, and z data for the sphere
% r = 80 * ones(50, 50); % radius is 5
% [th, phi] = meshgrid(linspace(0, 2*pi, 50), linspace(-pi, pi, 50));
% [x,y,z] = sph2cart(th, phi, r);
% x = x + p(j,1);  % center at 16 in x-direction
% y = y + p(j,2);  % center at 40 in y-direction
% z = z + p(j,3);   % center at 2 in z-direction
% % Let's say that this is how we make the existing 3D plot
% surf(peaks);
% % Now we use the surface command to add the sphere. We just need to set the FaceColor as desired.
% surface(x,y,z,'FaceColor', 'none')
% end

function shortlist1=reach(p,k)
i=0;
[row,col]=size(p);
 scatter3(p(k,1),p(k,2),p(k,3),80,'filled','g')
shortlist1=[];
 for j =1:row
     % reach_sphere(p,j);
     if distance(p,j,k)<50

        i=i+1;
        
         
            scatter3(p(j,1),p(j,2),p(j,3),20,'filled','b')
            
           
         
        i
     end
     drawnow
 end
 while k< row
     k=k+100;
     reach(p,k);
 end 
end
