// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract MedicalReputation {
    mapping(address => uint256) public scores;
    address public serverAdmin;

    constructor() {
        serverAdmin = msg.sender; 
    }

    function updateScore(address hospital, uint256 accuracy) public {
        require(msg.sender == serverAdmin, "Only server can update scores");
        
        if (accuracy > 80) {
            scores[hospital] += 10;
        } else {
            scores[hospital] += 2; 
        }
    }
}