cd ../src || exit
find -type f -regex '.*\.[ch]\(pp\)?' | xargs wc -l | sort -n
