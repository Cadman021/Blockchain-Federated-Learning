// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract MedicalReputation {
    struct ClientData {
        uint256[] history;
        uint256 currentReputation;
        bool isRegistered;
    }

    mapping(address => ClientData) public clients;
    address public owner;

    constructor() {
        owner = msg.sender;
    }

    // ثبت امتیاز جدید و به‌روزرسانی خودکار اعتبار
    function updateScore(address _client, uint256 _score) public {
        if (!clients[_client].isRegistered) {
            clients[_client].isRegistered = true;
            clients[_client].currentReputation = 50; // امتیاز اولیه برای نودهای جدید
        }

        clients[_client].history.push(_score);

        // محاسبه اعتبار به عنوان میانگین ۵ امتیاز آخر (Moving Average)
        uint256 count = clients[_client].history.length;
        uint256 start = count > 5 ? count - 5 : 0;
        uint256 sum = 0;
        uint256 actualCount = 0;

        for (uint256 i = start; i < count; i++) {
            sum += clients[_client].history[i];
            actualCount++;
        }

        clients[_client].currentReputation = sum / actualCount;
    }

    // دریافت اعتبار فعلی کلاینت برای وزن‌دهی در سرور
    function getReputation(address _client) public view returns (uint256) {
        if (!clients[_client].isRegistered) return 0;
        return clients[_client].currentReputation;
    }
}