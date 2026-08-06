#!/bin/sh -l

echo "MLPCpp Docker Container"
usage="$(basename "$0") [-h] [-b branch_name]
where:
    -h  show this help text
    -b  branch name (if not given, existing MLPCpp directory must be mounted in /src/MLPCpp).

Compiled binaries can be found at /install/. Mount that directory for access.
Note: If you specify a working directory using the --workdir option for docker,
      append this directory to all paths above (e.g. use --workdir=/tmp if running in user mode)."

flags=""
branch="main"
testscript=""
mlpcppb=""
workdir=$PWD
regression=false
unittest=false
export CCACHE_DIR=$workdir/ccache

if [ "$#" -ne 0 ]; then
  while [ "$(echo $1 | cut -c1)" = "-" ]
    do
        case "$1" in
            -R)
                    regression=true
                    shift 1
                ;;
            -U)
                    unittest=true 
                    shift 1
                ;;
            -b)
                    branch=$2
                    shift 2
                ;;
            *)
                    echo "$usage" >&2
                    exit 1
                ;;
    esac
    done
fi


name="MLPCpp_$(echo $branch | sed 's/\//_/g')"
echo "Branch provided. Cloning to $PWD/src/$name"
if [ ! -d "src" ]; then
  mkdir "src"
fi
cd "src"
git clone --recursive https://github.com/EvertBunschoten/MLPCpp $name
cd $name
git config --add remote.origin.fetch '+refs/pull/*/merge:refs/remotes/origin/refs/pull/*/merge'
git config --add remote.origin.fetch '+refs/heads/*:refs/remotes/origin/refs/heads/*'
git fetch origin
git checkout $branch

if [ "$unittest" = true ]; then 
cd UnitTests 
cmake -B build 
cd build
make
ctest --output-on-failure
fi 

if [ "$regressiontest" = true ]; then 

cd RegressionTests 
cmake -B build
cmake --build build 
./build/run_unit_tests
fi 