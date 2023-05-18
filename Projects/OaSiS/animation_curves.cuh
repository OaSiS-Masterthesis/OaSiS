#ifndef ANIMATION_CURVES_H
#define ANIMATION_CURVES_H

#include "settings.h"

struct Rest{
	mn::vec3 operator()(const mn::Duration& curr_time, const mn::Duration& dt) const noexcept{
		return mn::vec3{0.0f, 0.0f, 0.0f};
	}
};

struct Gravity{
	const float mass;
	
	Gravity(float mass)
	:mass(mass)
	{}
	
	mn::vec3 operator()(const mn::Duration& curr_time, const mn::Duration& dt) const noexcept{
		return mn::vec3{0.0f, mass * mn::config::G_GRAVITY, 0.0f};
	}
};

struct UpAndDown{
	const float range_start;
	const float range_end;
	const float init;
	const float speed;
	
	UpAndDown(float range_start, float range_end, float init, float speed)
	:range_start(range_start), range_end(range_end), init(init), speed(speed)
	{}
	
	mn::vec3 operator()(const mn::Duration& curr_time, const mn::Duration& dt) const noexcept{
		const float distance = std::abs(range_end - range_start);
		const float abs_init = init - range_start;//Map to range from 0 to distance
		
		const float abs_force_start_tmp = std::fmod((abs_init + speed * curr_time.count()), 2 * distance);
		const float abs_force_start = (abs_force_start_tmp > distance ? distance - (abs_force_start_tmp - distance)  : abs_force_start_tmp) ;
		
		const float abs_force_end_tmp = std::fmod((abs_init + speed * (curr_time + dt).count()), 2 * distance);
		const float abs_force_end = (abs_force_end_tmp > distance ? distance - (abs_force_end_tmp - distance)  : abs_force_end_tmp) ;
		
		const float abs_force = (abs_force_start + abs_force_end) * 0.5f;
		const float force = range_start + abs_force;
		
		return mn::vec3{0.0f, force, 0.0f};
	}
};

struct RotateAroundY{
	const float magnitude;
	const float time_start;
	const float time_end;
	
	RotateAroundY(float magnitude, float time_start, float time_end)
	:magnitude(magnitude), time_start(time_start), time_end(time_end)
	{}
	
	mn::vec3 operator()(const mn::Duration& curr_time, const mn::Duration& dt) const noexcept{
		if(curr_time.count() >= time_start && curr_time.count() <= time_end){
			return mn::vec3{0.0f, magnitude, 0.0f};
		}else{
			return mn::vec3{0.0f, 0.0f, 0.0f};
		}
	}
};

//TODO: Wrapper to combine animations and add starttime/endtime and such.


#endif