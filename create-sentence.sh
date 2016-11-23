
for d in */ ; do
    d=${d%%-*}
    printf "$d; "

    if [ ${d:0:1} = "s" ]
    then
	command="set "
    elif [ ${d:0:1} = "p" ];then
        command="place "
    elif [ ${d:0:1} = "l" ];then
        command="lay "
    elif [ ${d:0:1} = "b" ];then
        command="bin "
    fi

    if [ ${d:1:1} = "w" ]
    then
	color="white "
    elif [ ${d:1:1} = "g" ];then
        color="green "
    elif [ ${d:1:1} = "b" ];then
        color="blue "
    elif [ ${d:1:1} = "r" ];then
        color="red "
    fi

    if [ ${d:2:1} = "a" ]
    then
	pre="at "
    elif [ ${d:2:1} = "b" ];then
        pre="by "
    elif [ ${d:2:1} = "w" ];then
        pre="with "
    elif [ ${d:2:1} = "i" ];then
        pre="in "
    fi
    #letter
    letter="${d:3:1} " 
    #digit
    if [ ${d:4:1} = "z" ]
    then
        digit="zero "
    elif [ ${d:4:1} = "1" ];then
        digit="one "
    elif [ ${d:4:1} = "2" ];then
        digit="two "
    elif [ ${d:4:1} = "4" ];then
        digit="four "
    elif [ ${d:4:1} = "5" ];then
        digit="five "
    elif [ ${d:4:1} = "6" ];then
        digit="six "
    elif [ ${d:4:1} = "7" ];then
        digit="seven "
    elif [ ${d:4:1} = "8" ];then
        digit="eight "
    elif [ ${d:4:1} = "9" ];then
        digit="nine "
    fi

    if [ ${d:5:1} = "a" ]
    then
	adv="soon"
    elif [ ${d:5:1} = "p" ];then
        adv="please"
    elif [ ${d:5:1} = "n" ];then
        adv="now"
    elif [ ${d:5:1} = "a" ];then
        adv="again"
    fi
    sentence=$command$color$pre$letter$digit$adv 
    size=${#sentence}
    #echo $size"; "$sentence
    #translate
    foo=$sentence
    encode=""
    for (( i=0; i<${#foo}; i++ )); do
        #printf %02X \'${foo:$i:1}; printf " "
        tmp=$(printf '%d' "'${foo:$i:1}"); 
        tmp=$((tmp-30))
        #if the character transfer to not equal to 2, it means not a blank, and we need to deduct more
        if [ $tmp -ne 2 ]
	then
		tmp=$((tmp-64))
	fi
        encode=$encode" "$tmp
        #printf ";"$tmp" "
    done
    echo $d"; "$size"; "$sentence"; "$encode >> training-table.txt
    #${d:0:1}
done
