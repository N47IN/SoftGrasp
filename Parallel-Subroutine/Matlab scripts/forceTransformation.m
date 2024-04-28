% R Navin Sriram, n47insriram@gmail.com
% For a 3 jawed gripper
% Importing and creating a mesh from a STL file object
% uncomment all scatter3 to get 



model = createpde;
impGeo = importGeometry(model,'mesh.stl');
mesh= generateMesh(model,"Hmin",50);
[p,edges,t]=meshToPet(mesh);
m=0.2;
g=9.8;
magnitude = 5;

x = p(1,:);
y = p(2,:);
z = p(3,:);

x_com = mean(x);
y_com = mean(y);
z_com = mean(z);

P= p';
Tx = [1 0 0 -x_com;0 1 0 -y_com;0 0 1 -z_com;0 0 0 1];
P(:,4)=ones(size(P(:,1)));
req_P = Tx*P';

% % Through Convex Hull
% 
% K1 = convhull(p(1,:),p(2,:),p(3,:));
% subplot(1,2,1)
% defaultFaceColor  = [0.6875 0.8750 0.8984];
% trisurf(K1,p(1,:),p(2,:),p(3,:),'Facecolor',defaultFaceColor)
% axis equal
 

% Delaunay triangulation for triangulating the outer surface

DT = delaunayTriangulation(req_P(1,:)',req_P(2,:)',req_P(3,:)');
[T,Xb] = freeBoundary(DT);
TR = triangulation(T,Xb); 
G_pts = incenter(TR);
G_norm = faceNormal(TR);  

ext_Wrench = [ 0; 0; -m*g; 0 ; 0; 0; ] ;

% trisurf(T,Xb(:,1),Xb(:,2),Xb(:,3), ...
%      'FaceColor','cyan','FaceAlpha',0.8);
axis equal
hold on  
scatter3(G_pts(:,1),G_pts(:,2),G_pts(:,3),10,'filled','r');

k=200;

Filter = reach3(G_pts,k);
Grasp = transf(Filter, G_norm, G_pts, magnitude);
Optim = rlse(Grasp,ext_Wrench);

% linear least square optimisation
% function handle = rlse(shortlist, ext_Wrench,k)
%  [rows,cols]=size(shortlist.Handle);
%  handle = linspace(1,rows,rows-1);
%  comb = nchoosek(handle,2);
% end




%Adjoint Map from transforming the wrench from grasp point frame to COM
%frame


function shortlist = transf(shortlist, G_norm, G_pts, magnitude)

 [row,col] = size (shortlist.X);
 shortlist.Gmaps=[];
 shortlist.handForce=[];


 for j = 1:col
   h = shortlist.Handle(j);
   G_frc= magnitude * G_norm(h,:) / norm(G_norm(h,:));
   tau_x = sqrt((G_pts(h,1)).^2 + (G_pts(h,3)).^2)*G_frc(1,2) + sqrt((G_pts(h,1)).^2 + (G_pts(h,2)).^2)*G_frc(1,3);
   tau_y = sqrt((G_pts(h,2).^2) + (G_pts(h,3).^2))*G_frc(1,1) + sqrt((G_pts(h,2).^2) + (G_pts(h,1).^2))*G_frc(1,3);
   tau_z = sqrt((G_pts(h,3).^2) + (G_pts(h,1).^2))*G_frc(1,2) + sqrt((G_pts(h,3).^2) + (G_pts(h,2).^2))*G_frc(1,1);

   G_frc(1,4) = tau_x;
   G_frc(1,5) = tau_y;
   G_frc(1,6) = tau_z;

   shortlist.handForce(end+1,:)= G_frc(1,:);
 
 end

end

function dist=distance(x,k,j)
 dist=sqrt((x(k,1)-x(j,1)).^2 + (x(k,2)-x(j,2)).^2 + (x(k,3)-x(j,3)).^2);
end

% function rank=rank(C,G_norm)
%  [row,col]=size(C);
%  for i =1:col
%      rank_mat=[G_norm(200,:);G_norm(C(i,1),:) ;G_norm(C(i,2),:)];
%      if det(rank_mat)~=0
%          C(i)=[];
%      end
%  end
%  rank = C;
% end


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


function shortlist=reach3(pt,k)

 shortlist=struct();
 shortlist.X=[];
 shortlist.Y=[];
 shortlist.Z=[];
 shortlist.Handle=[];

 i=0;
 [row,col]=size(pt);
 %scatter3(p(k,1),p(k,2),p(k,3),80,'filled','g');
 for j =1:row
     % reach_sphere(p,j);
     if distance(pt,j,k)<50

        i=i+1;
        %scatter3(p(j,1),p(j,2),p(j,3),30,'filled','b')
     
        shortlist.X(end+1)=pt(j,1);
        shortlist.Y(end+1)=pt(j,2);
        shortlist.Z(end+1)=pt(j,3);
        shortlist.Handle(end+1)=j;
     end
     
 end  
 
end




