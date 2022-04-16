#ifndef H_BOUNDARY
#define H_BOUNDARY

//supported boundary conditions
enum class B_CONDITION {
	INTERIOR,
	NO_SLIP,
	PERIODIC_UL,
	PERIODIC_LR,
};

#endif
