function getBotResponse(input) {
    //rock paper scissors
    if (input == "How are you?") {
        return "Good! yourself?";
    } else if (input == "Can I ask you something?") {
        return "Sure! what is your query?";
    } else if (input == "Good bye") {
        return "Have a nice day!";
    }

    // Simple responses
    if (input == "hello") {
        return "Hello there!";
    } else if (input == "goodbye") {
        return "Talk to you later!";
    } else {
        //return "Sorry! can you try asking something else!?";
        return "How can I help you today?";
    }
}