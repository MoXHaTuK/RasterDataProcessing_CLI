#pragma once
#include "ImageManager.hpp"
#include <cuda_runtime.h>

namespace GPU {

	float launchAdd(const Image&, const Image&, Image&);
	float launchSub(const Image&, const Image&, Image&);
	float launchSmooth(const Image&, Image&);
	float launchEnhance(const Image&, Image&);
	float launchErode(const Image&, Image&);
	float launchDilate(const Image&, Image&);

}
