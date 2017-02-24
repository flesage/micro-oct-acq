x=[1313,588];
lam_min=x(1)-x(2)/2;
lam_max=x(1)+x(2)/2;
nln_k=2*pi./linspace(lam_max,lam_min,2048);
lin_k=linspace(min(nln_k),max(nln_k),2048);
int_mat=zeros(2048,2048);
for i=1:2048
    tmp=zeros(2048,1);
    tmp(i)=1;   
    int_mat(:,i)=interp1(nln_k,tmp,lin_k);
    
end

f=fopen('interpolation_matrix.dat','wb');
fwrite(f,int_mat,'double');
fclose(f);

x=[1310,344];
lam_min=x(1)-x(2)/2;
lam_max=x(1)+x(2)/2;
nln_k=2*pi./linspace(lam_max,lam_min,2048);
lin_k=linspace(min(nln_k),max(nln_k),2048);
int_mat=zeros(2048,2048);
for i=1:2048
    tmp=zeros(2048,1);
    tmp(i)=1;   
    int_mat(:,i)=interp1(nln_k,flipud(tmp),lin_k);
    
end

f=fopen('interpolation_matrix_cobra.dat','wb');
fwrite(f,int_mat,'double');
fclose(f);