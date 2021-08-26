using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Text;

namespace SuaveKeys.SnapReader.Uwp.Models
{
    public class TokenResponse
    {
        [JsonProperty("access_token")]
        public string AccessToken { get; set; }
        [JsonProperty("refresh_token")]
        public string RefreshToken { get; set; }
        [JsonProperty("access_token_expiration")]
        public DateTime AccessTokenExpiration { get; set; }
        [JsonProperty("refresh_token_expiration")]
        public DateTime RefreshTokenExpiration { get; set; }
        public string UserId { get; set; }
    }
}
