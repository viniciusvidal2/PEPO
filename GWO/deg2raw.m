function [res]=deg2raw(deg,motor)
deg_min_tilt = 28;
deg_hor_tilt = 0;
deg_max_tilt = -60.9;
raw_deg = 11.37777;
deg_raw = 1 / raw_deg;
if(strcmp(motor,'pan'))
    res= deg*raw_deg;
elseif(strcmp(motor,'tilt'))
    res =(deg - deg_min_tilt)*raw_deg + raw_min_tilt;
end
end
