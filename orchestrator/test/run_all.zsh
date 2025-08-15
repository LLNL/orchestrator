#!/bin/zsh

# Execute this script to automatically run all of the provided tests
# note that this script should be executed once the correct environment has
# been loaded, and likely from an interactive job on a compute node

for dir in $(find . -name "driver.py" -type f | cut -d'/' -f 1,2); do
    cd $dir
    echo "################### Now running tests for $dir ###################"
    if [[ -f orch.log ]]; then
        echo "It looks like these were already run, skipping automatic submit"
    else
        python driver.py
    fi
    echo "################### Finished tests for $dir ###################"
    echo
    cd ..
done
echo "Finished running all tests!"
echo
