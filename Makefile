# Makefile
#
# Rules:
# all		-> build
# build		build the program
# clean		clean built files

-include Makefile.user


# Project name
NAME=iblgf.x
SOURCES = main.cpp 
PREPROC=junk

# Dependency directory
DEPDIR = .deps
df = $(DEPDIR)/$(*F)

# Compiler flags settings
CXX ?= mpic++

CFLAGS = -fconstexpr-depth=2048 \
		 -mavx \
		 -std=c++17 \
		 -Wall 
CFLAGS_DEBUG = -O0 -ggdb 

CFLAGS_RELEASE = -O3 -DNDEBUG \
				 -DBOOST_DISABLE_ASSERTS \
				 -march=native  \
				 -Wno-unused-local-typedefs  \
				 -Wno-deprecated-declarations

CFLAGS_UNSAFE = -ffast-math -ftree-vectorize -funroll-loops

INCLUDE_DIRS = $(USER_INCLUDE_DIRS) \
			   -I./ \
			   -I./dictionary \
			   -I./domain \
			   -I./domain/octree \
			   -I./domain/dataFields \
			   -I./tensor 

LIB_DIRS = $(USER_LIB_DIRS) 
LDFLAGS = $(USER_LDFLAGS) \
		  -lboost_mpi \
		  -lboost_serialization \
		  -lboost_system \
		  -lboost_filesystem

OBJFILES = $(SOURCES:.cpp=.o)

#all: CFLAGS += $(CFLAGS_RELEASE)
all: build

debug: CFLAGS += $(CFLAGS_DEBUG)
debug: build

release: CFLAGS += $(CFLAGS_RELEASE)
release: build

unsafe: CFLAGS += $(CFLAGS_RELEASE)
unsafe: CFLAGS += $(CFLAGS_UNSAFE)
unsafe: build

profile: CFLAGS += $(CFLAGS_RELEASE)
profile: CFLAGS += -g -pg
profile: LDFLAGS += -pg
profile: build

preprocess: $(SOURCES) 
	@echo -e "\033[1mPreprocessing $(SOURCES) to $(PREPROC)...\033[0m"
	@$(CXX) $(CFLAGS) $(INCLUDE_DIRS) -E $(SOURCES) > $(PREPROC)

build: $(OBJFILES) 
	@echo -e "\033[1mLinking $(OBJFILES) to $(NAME)...\033[0m"
	@$(CXX) $(CFLAGS) $(INCLUDE_DIRS) $(OBJFILES) -o $(NAME) $(LDFLAGS)

%.o: %.cpp
	@mkdir -p .deps
	@echo -e "\033[1mCompiling $<...\033[0m"
	$(CXX) -c $(CFLAGS) $(INCLUDE_DIRS) -MD $< -o $@
	@cp $*.d $(df).P; \
	sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
	    -e '/^$$/ d' -e 's/$$/ :/' < $*.d >> $(df).P; \
	rm -f $*.d;

-include $(SOURCES:%.cpp=$(DEPDIR)/%.P)

clean:
	@echo -e "\033[1mCleaning up...\033[0m"
	@rm -f $(OBJFILES)
	@rm -f $(NAME)
	@rm -f $(PREPROC)

.PHONY: all debug release unsafe profile build clean

